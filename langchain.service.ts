import { Injectable, OnModuleInit, Logger, Inject } from '@nestjs/common';
import { DataSource } from 'typeorm';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector'; // Adjust the path based on documentation
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { DocumentInterface } from '@langchain/core/documents';
import { z } from 'zod';
import { StringOutputParser } from '@langchain/core/output_parsers';

interface QuestionAnswerWithSource {
  question?: string;
  answer?: string;
  sources?: { pageContent?: string; source?: string }[];
}

@Injectable()
export class LangService implements OnModuleInit {
  private pgVectorStore: PGVectorStore;
  private readonly logger = new Logger(LangService.name);

  constructor(
    @Inject('DATA')
    private dataSource: DataSource,
  ) {}

  async onModuleInit() {
    try {
      const embeddings = new OpenAIEmbeddings({
        model: 'text-embedding-3-small',
      });

      const con: any = this.dataSource.options;

      this.pgVectorStore = new PGVectorStore(embeddings, {
        postgresConnectionOptions: {
          host: con.host,
          port: con.port,
          user: con.username,
          password: con.password,
          database: con.database,
        },
        tableName: 'qa_embeddings',
        schemaName: 'public',
        columns: {
          idColumnName: 'id',
          vectorColumnName: 'embedding',
          contentColumnName: 'content',
          metadataColumnName: 'metadata',
        },
        chunkSize: 500,
        verbose: true,
      });

      this.logger.log('Vector store initialized and table structure ensured.');
    } catch (error) {
      this.logger.error('Failed to initialize vector store:', error);
      throw error;
    }
  }

  async createEmbeddings(
    data: {
      id: string;
      text: string;
      tenantId: string;
      source: string;
      companyId?: string;
      userTenantRoleId?: string;
      answeredAt?: string;
    }[],
  ) {
    try {
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });

      for (const batch of this.chunkArray(data, 10)) {
        const documents: any[] = [];
        for (const item of batch) {
          const splits = await splitter.splitDocuments([
            { id: item.id, pageContent: item.text, metadata: {} },
          ]);
          this.logger.log(
            `Document ID ${item.id} split into ${splits.length} chunks.`,
          );

          let chunkNumber = 0;
          for (const split of splits) {
            documents.push({
              id: `${item.id}-${chunkNumber}`,
              pageContent: split.pageContent,
              metadata: {
                tenantId: item.tenantId,
                source: item.source,
                answeredAt: item.answeredAt,
                chunkNumber: chunkNumber++,
                chunkParentId: item.id,
                ...(item.companyId && { companyId: item.companyId }),
                ...(item.userTenantRoleId && {
                  userTenantRoleId: item.userTenantRoleId,
                }),
              },
            });
          }
        }
        await this.pgVectorStore.addDocuments(documents);
        this.logger.log('Batch embeddings created successfully.');
      }
      return 'Batch embeddings created successfully.';
    } catch (error) {
      this.logger.error(
        'Error creating embeddings from document splits:',
        error,
      );
      throw error;
    }
  }

  async searchMostSimilarDocument(
    query: string,
    filters: {
      tenantId: string;
      source?: string;
      companyId?: string;
      userTenantRoleId?: string;
      questionnaireId?: string;
    },
    topK: number = 5,
    modelType: string = 'text-embedding-3-small',
  ) {
    try {
      const model = new OpenAIEmbeddings({ model: modelType });

      const improvedSemanticQuery = await this.transformQuery(query);
      this.logger.log(`ImprovedSemanticQuery: ${improvedSemanticQuery}`);
      const vectorQuery = await model.embedQuery(improvedSemanticQuery);

      const searchResult =
        await this.pgVectorStore.similaritySearchVectorWithScore(
          vectorQuery,
          topK,
          filters,
        );

      if (!searchResult || searchResult.length === 0) {
        this.logger.warn(
          'No matching document found for the query and specified filters.',
        );
        return null;
      }

      this.logger.log(
        `Search completed successfully. Results: ${JSON.stringify(
          searchResult.map((res) => ({ id: res[0].id, score: res[1] })),
        )}`,
      );

      //const gradedDocuments = await this.gradeDocuments(query, searchResult);

      return this.formulateAnswer(query, searchResult);
    } catch (error) {
      this.logger.error('Error performing search with filters:', error);
      throw error;
    }
  }

  /**
   * Determines whether the retrieved documents are relevant to the question.
   */
  async gradeDocuments(
    query: string,
    documents: [DocumentInterface, number][],
  ): Promise<[DocumentInterface, number][]> {
    console.log('---CHECK RELEVANCE---');
    const model = new ChatOpenAI({
      model: 'gpt-4o',
      temperature: 0,
    });

    const llmWithTool = model.withStructuredOutput(
      z
        .object({
          binaryScore: z
            .enum(['yes', 'no'])
            .describe("Relevance score 'yes' or 'no'"),
        })
        .describe(
          "Grade the relevance of the retrieved documents to the question. Either 'yes' or 'no'.",
        ),
      {
        name: 'grade',
      },
    );

    const prompt = ChatPromptTemplate.fromTemplate(
      `You are a grader assessing relevance of a retrieved text to a user question. The text will have the question and then the answer.
  Here is the retrieved text:
  
  {context}
  
  Here is the user question: {question}

  If the text contains any keyword(s) or semantic meaning related to the user question, grade it as relevant.
  Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.`,
    );

    const chain = prompt.pipe(llmWithTool);

    const filteredDocs: [DocumentInterface, number][] = [];
    for await (const doc of documents) {
      const grade = await chain.invoke({
        context: doc[0].pageContent,
        question: query,
      });
      if (grade.binaryScore === 'yes') {
        console.log('---GRADE: DOCUMENT RELEVANT---');
        filteredDocs.push(doc);
      } else {
        console.log('---GRADE: DOCUMENT NOT RELEVANT---');
      }
    }
    return filteredDocs;
  }

  async formulateAnswer(
    query: string,
    documents: [DocumentInterface, number][],
  ): Promise<QuestionAnswerWithSource> {
    const model = new ChatOpenAI({
      model: 'gpt-4o',
    });

    const llmWithTool = model.withStructuredOutput(
      z
        .object({
          question: z.string().describe('The question to answer'),
          answer: z.string().describe('Answer for the question'),
          sources: z.array(
            z
              .object({
                pageContent: z.string().describe('The source answer'),
                source: z
                  .string()
                  .describe('The source to answer from retrieved text'),
              })
              .describe('Source to answer from retrieved text'),
          ),
        })
        .describe(
          'Answer the question based on the retrieved documents related to the question.',
        ),
      {
        name: 'question_answer',
      },
    );

    const prompt = ChatPromptTemplate.fromTemplate(
      `You are a question answering assistant. Answer the question based on the retrieved text. The text will have the question and then the answer.
  Here is the retrieved text:
  
  {context}
  
  Here is the question: {question}

  Don't answer the question if the retrieved text doesn't have the answer. Don't include the question in the answer. Keep it short and precises.
  Return the question and the answer. Add the source from the retrieved text if it's been used to answer the question`,
    );

    return prompt.pipe(llmWithTool).invoke({
      context: documents,
      question: query,
    });
  }

  /**
   * Transform the query to produce a better question.
   */
  async transformQuery(question: string): Promise<string> {
    console.log('---TRANSFORM QUERY---');

    // Pull in the prompt
    const prompt = ChatPromptTemplate.fromTemplate(
      `You are generating a question that is well optimized for semantic search retrieval.
  Look at the input and try to reason about the underlying sematic intent / meaning.
  Here is the initial question:
  \n ------- \n
  {question} 
  \n ------- \n
  Formulate an improved question: `,
    );

    const model = new ChatOpenAI({
      model: 'gpt-4o',
      temperature: 0,
    });

    // Construct the chain
    const chain = prompt.pipe(model).pipe(new StringOutputParser());
    return chain.invoke({ question });
  }

  private chunkArray<T>(array: T[], size: number): T[][] {
    return Array.from({ length: Math.ceil(array.length / size) }, (_, i) =>
      array.slice(i * size, i * size + size),
    );
  }
}
